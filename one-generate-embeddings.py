import os
import sys
import numpy as np
import io
from sentence_transformers import SentenceTransformer, models
from tqdm import tqdm
import csv

def generate_embeddings(model, output_dir: str, dataset_num: int):
    """
    Encode title and body separately, concatenate the vectors, and save each
    embedding to test-<dataset_num>/<output_dir>/<doc_id>_tb.npy.

    Returns (num_rows_processed, dest_dir)
    """
    dest_dir = os.path.join(f"test-{dataset_num}", output_dir)
    os.makedirs(dest_dir, exist_ok=True)

    csv_path = f"datasets/v{dataset_num}_sample_data.csv"

    csv_length = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in tqdm(reader, desc="Processing documents"):
            csv_length += 1                           
            doc_id = row["doc_id"].strip()
            title  = row["title"].strip()
            body   = row["body"].strip()

            title_emb = model.encode(title)
            body_emb  = model.encode(body)
            embedding = np.concatenate([title_emb, body_emb])

            buffer = io.BytesIO()
            np.save(buffer, embedding)
            buffer.seek(0)

            file_path = os.path.join(dest_dir, f"{doc_id}_tb.npy")
            with open(file_path, "wb") as out_f:
                out_f.write(buffer.read())

    return csv_length, dest_dir

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

    dataset_num = 1
    num_embeddings, output_dir = generate_embeddings(model=model, output_dir=output_dir, dataset_num=dataset_num)
    print(f"Generated {num_embeddings} embeddings in directory: {output_dir}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python3 one-generate-embeddings.py <model_num>")
        sys.exit(1)

    model_num = sys.argv[1]
    main(model_num=model_num)
