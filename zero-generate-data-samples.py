import os
import pandas as pd
from sqlconnector import getconn

def save_sample_data(num_docs: int, file_name: str) -> str:

    # retrieve data from the database
    with getconn() as conn:
        cur = conn.cursor()

        bernoulli_randomizer = 0.9

        cur.execute(f"""
            SELECT doc_id, body
            FROM docs TABLESAMPLE BERNOULLI({bernoulli_randomizer})
            WHERE array_length(regexp_split_to_array(body, '\\s+'), 1) BETWEEN 10 AND 150
            LIMIT 100000;
        """)
        
        rows = cur.fetchall()

    # randomly sample num_docs documents
    df = pd.DataFrame(rows, columns=["doc_id", "body"])

    if len(df) < num_docs:
        print(f"Warning: Only {len(df)} documents available. Sampling all of them.")
        sample_df = df
    else:
        sample_df = df.sample(n=num_docs, random_state=42)


    # save sample to specified filepath    
    os.makedirs("datasets", exist_ok=True)
    output_path = os.path.join("datasets", file_name)
    sample_df.to_csv(output_path, index=False)

    print(f"Saved {len(sample_df)} documents to {output_path}")

    return output_path


def main():
    
    num_docs = 10000
    file_name = "v3_sample_data.csv"

    save_sample_data(num_docs, file_name)

if __name__ == "__main__":
    main()