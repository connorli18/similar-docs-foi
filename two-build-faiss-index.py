import faiss
import numpy as np
import os
from tqdm import tqdm
import sys


def build_faiss_index(embeddings_dir: str, output_dir: str, model_name: str) -> list:
    """
    Takes embeddings and builds a FAISS index to store them. 
    Also, saves the document IDs in a separate file to link them to the embeddings.
    (i.e. doc_ids[i] corresponds to embeddings[i])

    Args:
        - embeddings_dir: Directory containing the embeddings
        - output_dir: Directory to save the FAISS index and doc_id mapping
        - model_name: Model name (used in file naming)

    Returns:
        - List of file paths [FAISS index, doc_id_mapping]
    """

    # Store embedding files in a list
    embedding_files = [f for f in os.listdir(embeddings_dir) if f.endswith(".npy")]

    # Store document IDs and embeddings (not filenames) in separate lists
    doc_ids = []
    embeddings = []

    # Load embeddings and document IDs and store them in lists
    for file in tqdm(embedding_files, desc="Processing embeddings"):
        doc_id = os.path.splitext(file)[0]  
        embedding = np.load(os.path.join(embeddings_dir, file))

        doc_ids.append(doc_id)  
        embeddings.append(embedding)

    # Convert lists to numpy arrays (float32 required for FAISS)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Index object for efficient searching of embeddings
    dimension = embeddings.shape[1]
    print(dimension)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  

    # Save the FAISS index and doc_id mapping
    os.makedirs(output_dir, exist_ok=True)
    faiss_index_path = os.path.join(output_dir, f"{model_name}_faiss_index.bin")
    doc_id_mapping_path = os.path.join(output_dir, f"{model_name}_doc_ids.npy")

    faiss.write_index(index, faiss_index_path)
    np.save(doc_id_mapping_path, np.array(doc_ids))

    return [faiss_index_path, doc_id_mapping_path]


def build_faiss_wrapper(model_num: str, test_set: int):

    OUTPUT_DIR = f"test-{test_set}/faiss-storage"

    # Single dictionary containing both model name and embeddings directory
    model_map = {
        "1": ("mini_lm", "mini-lm-embeddings"),
        "2": ("longformer", "longformer-embeddings"),
        "3": ("msmarco_bert", "msmarco-bert-embeddings")
    }

    # Check if the model_num is valid
    if model_num not in model_map:
        print(f"Error: Invalid model_num '{model_num}'. Choose from {list(model_map.keys())}.")
        sys.exit(1)

    # Unpack tuple values from dictionary
    model_name, embeddings_dir = model_map[model_num]
    embeddings_dir = f"test-{test_set}/{embeddings_dir}"

    # Call build_faiss_index to generate the FAISS index and document ID mapping
    faiss_index, doc_id_mapping = build_faiss_index(
        embeddings_dir=embeddings_dir,
        output_dir=OUTPUT_DIR,
        model_name=model_name
    )

    # Output the results
    print(f"FAISS index saved to: {faiss_index}")
    print(f"Document ID mapping saved to: {doc_id_mapping}")

def main():

    # Accept argv for the test_set
    if len(sys.argv) < 2:
        print("Usage: python two-build-faiss-index.py <test_set>")
        sys.exit(1)

    test_set = int(sys.argv[1])  # Convert test_set to an integer

    for model_num in ["1", "2", "3"]:
        print(f"Building FAISS index for test set {test_set} model {model_num}...")
        build_faiss_wrapper(model_num=model_num, test_set=test_set) 

if __name__ == "__main__":
    main()