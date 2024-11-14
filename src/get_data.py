import logging
import arxiv
import argparse
from itertools import chain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# init the logger 
logger = logging.getLogger(__name__)

# temp list of documents to leverage
PAPER_IDS = [
    "1706.03762", # attention is all you need
    "2411.04952", # M3DOCRAG
    "2106.09685", # Lora
    "2407.01449", # Colpali
    "2003.05622",
    "2203.15556" # Chinchilla paper 
]

PAPER_SEARCH_KEYS = [
    "llm"
]

def main(max_paper: int) -> None:
    
    # init the default arxiv client 
    client = arxiv.Client()
    
    # get the top `max_paper` to be downloaded 
    top_papers_search = arxiv.Search(
        query = PAPER_SEARCH_KEYS[0], # only leverage the first keyword
        max_results = max_paper,
        sort_by = arxiv.SortCriterion.SubmittedDate
    )
    top_papers = client.results(top_papers_search)
    top_papers_ids = [paper.entry_id.split('/')[-1] for paper in top_papers] # example: 'http://arxiv.org/abs/2411.07133v1' -> 2411.07133v1
    
    # get the list of all papers to download 
    all_papers_ids = list(chain(top_papers_ids, PAPER_IDS))
    
    # download the papers:
    for paper_id in all_papers_ids:
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[paper_id])))
        
        # dowload
        paper.download_pdf(
            dirpath="data/papers", # turn this into a script argument
            filename=f"{paper_id}.pdf"
        )
        logger.info(f"Paper {paper_id} has been successfully downloaded")
    
    logger.info("All papers have been successfully downloaded")


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(
        description="Download papers to build the assistant knowledge base"
    )
    
    # add the max number of documents to search 
    parser.add_argument(
        "--max_paper_to_search",
        help="define the max number of parameters to retrieve from Arxiv",
        type=str
    )
    
    # get the args 
    args = parser.parse_args()
    
    # max number of papers to search 
    max_paper_to_search = int(args.max_paper_to_search)
    main(
        max_paper=max_paper_to_search
    )