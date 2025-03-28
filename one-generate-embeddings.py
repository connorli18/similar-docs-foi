import os
import sys
import numpy as np
import io
from sentence_transformers import SentenceTransformer, models
from tqdm import tqdm
import csv

def generate_embeddings(model, output_dir: str, dataset_num: int) -> list:
    
    output_dir = os.path.join(f"test-{dataset_num}", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load the sample data
    with open(f"datasets/v{dataset_num}_sample_data.csv", "r", encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) 
        csv_length = 0

        # Generate embeddings for each document
        for doc_id, text in tqdm(reader, desc="Processing documents"):
            csv_length += 1
            
            embedding = model.encode(text)

            # Save the embeddings to disk in .npy format
            np_bytes = io.BytesIO()
            np.save(np_bytes, embedding)
            np_bytes.seek(0)

            file_name = os.path.join(output_dir, f"{doc_id}.npy")
            with open(file_name, "wb") as f:
                f.write(np_bytes.read())

    return [csv_length, output_dir]

def main(model_num: int):

    model_map = {
        "1": ("mini-lm-embeddings", "all-MiniLM-L6-v2"), # fastest - 264
        "2": ("longformer-embeddings", "allenai/longformer-base-4096"),  # 4096
        "3": ("msmarco-bert-embeddings", "sentence-transformers/msmarco-bert-base-dot-v5") # specifically for legal texts? 512
    }

    if model_num not in model_map:
        print(f"Error: Invalid model_num '{model_num}'. Choose from {list(model_map.keys())}.")
        sys.exit(1)

    output_dir, model_name = model_map[model_num]

    print(f"Loading model: {model_name}...")

    if model_name == "allenai/longformer-base-4096":
        word_embedding_model = models.Transformer(model_name, max_seq_length=4096)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    else:
        model = SentenceTransformer(model_name)

    dataset_num = 3
    num_embeddings, output_dir = generate_embeddings(model=model, output_dir=output_dir, dataset_num=dataset_num)
    print(f"Generated {num_embeddings} embeddings in directory: {output_dir}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python3 generate_embeddings.py <model_num>")
        sys.exit(1)

    model_num = sys.argv[1]
    main(model_num=model_num)
