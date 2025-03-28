import faiss
import numpy as np
import os
import sys
from pprint import pprint

FAISS_DIR = "faiss-storage" 

MODEL_MAP = {
    "1": ("mini_lm", "mini-lm-embeddings"),
    "2": ("longformer", "longformer-embeddings"),
    "3": ("msmarco_bert", "msmarco-bert-embeddings")
}

def find_similar_docs(test_set: int, model_prefix: str, embedding_dir: str, num_docs: int, doc_id: str) -> list:
    """
    Find the most similar documents to a given document ID using FAISS index-querying.
    
    Args:
        num_docs (int): Number of similar documents to retrieve.
        doc_id (str): The document ID to search for.
    
    Returns:
        List of similar document IDs.

    """
    FAISS_DIR = f"test-{test_set}/faiss-storage"
    embedding_dir = f"test-{test_set}/{embedding_dir}"

    # index: faiss index object (stores embeddings)
    index = faiss.read_index(os.path.join(FAISS_DIR, f"{model_prefix}_faiss_index.bin"))
    
    # doc_ids: links embeddings to document IDs
    doc_ids = np.load(os.path.join(FAISS_DIR, f"{model_prefix}_doc_ids.npy"), allow_pickle=True)  

    # find the embedding for the query document
    query_path = os.path.join(embedding_dir, f"{doc_id}.npy")
    if not os.path.exists(query_path):
        raise FileNotFoundError(f"Embedding file '{query_path}' not found!")

    # load the query embedding if the file exists
    query_embedding = np.load(query_path).reshape(1, -1).astype("float32")

    # search for most similar documents (+1 because query document is always I[0])
    # D: distance to the nearest neighbors
    # I: indices of the "nearest neighbors"
    D, I = index.search(query_embedding, num_docs+1) 

    # use the doc_ids mapping to connect indices to doc_ids
    similar_doc_ids = [doc_ids[i] for i in I[0] if i < len(doc_ids)] 

    #print(f"Top {num_docs} similar documents for '{doc_id}':", similar_doc_ids[1:])
    return similar_doc_ids[1:]

def find_docs_wrapper(test_set: int, doc_id: str, model_num: str) -> list:

    model_name, embeddings_dir = MODEL_MAP[model_num]
    return find_similar_docs(test_set=test_set, model_prefix=model_name, embedding_dir=embeddings_dir, num_docs=10, doc_id=doc_id)

def find_for_all_models(test_set: int, doc_id: str) -> dict:

    similar_results = {}

    for model_num in ["1", "2", "3"]:
        similar_docs = find_docs_wrapper(test_set=test_set, doc_id=doc_id, model_num=model_num)
        model_name, _ = MODEL_MAP[model_num]
        similar_results[model_name] = similar_docs
    
    return similar_results

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 three-find-sim-docs.py <doc_id> <test_set>")
        sys.exit(1)

    doc_id = sys.argv[1]
    test_set = int(sys.argv[2])

    results = find_for_all_models(test_set=test_set, doc_id=doc_id)
    pprint(results)


if __name__ == "__main__":
    ###### find . -maxdepth 1 -type f | awk 'BEGIN {srand()} {print rand(), $0}' | sort -n | cut -d" " -f2- | head -n 5
    main()
