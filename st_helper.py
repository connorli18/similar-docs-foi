import pandas
import random
import os
from three_find_sim_docs import find_for_all_models

DATASET_PREFIX = "datasets/"

def random_doc_select(dataset: str) -> str:

    dataset = os.path.join(DATASET_PREFIX, dataset)
    if not os.path.exists(dataset):
        raise FileNotFoundError(f"Dataset {dataset} does not exist.")

    with open(dataset, "r", encoding='utf-8') as f:
        reader = pandas.read_csv(f)
        doc_ids = reader["doc_id"].tolist()
        random_doc_id = random.choice(doc_ids)

        return str(random_doc_id)
    

def find_similar_docs(
    test_set: str,
    doc_id: str
) -> dict:
    
    test_set = int(str(test_set[1]))
    
    return find_for_all_models(
        test_set=test_set,
        doc_id=doc_id
    )