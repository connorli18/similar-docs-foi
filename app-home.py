import os
import streamlit as st
from st_helper import random_doc_select, find_similar_docs
import pandas as pd
from collections import OrderedDict

st.set_page_config(
    page_title="Find Similar Documents",
    page_icon=":mag_right:",
    layout="wide",
)

st.markdown("""
    <style>
    .stSelectbox, .stTextInput {
        max-width: 500px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("Find Similar Documents")
st.write(
    "This app allows you to find similar documents based on their embeddings. "
    "You can select a test set, randomly generate a document ID from that set, and view the retrieval results for the most similar documents."
)

def format_doc_links(doc_id, base_url="https://docviewer.history-lab.org/?doc_id="):
    #return f"<a href='{base_url}{doc_id}' target='_blank'>{doc_id}</a>"
    return f"<a href='{base_url}{doc_id}' target='_blank'>{doc_id}</a>"


def display_stats(results: dict) -> None:
    st.subheader("Document Overlap Statistics")
    st.write("This section provides statistics on the overlap of similar documents across the three models.")

    # Calculate Overlap between Models
    overlap_mini_lm_longformer = set(results["mini_lm"]).intersection(set(results["longformer"]))
    overlap_mini_lm_msmarco_bert = set(results["mini_lm"]).intersection(set(results["msmarco_bert"]))
    overlap_longformer_msmarco_bert = set(results["longformer"]).intersection(set(results["msmarco_bert"]))
    overlap_all_models = set(results["mini_lm"]).intersection(set(results["longformer"]), set(results["msmarco_bert"]))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader("MiniLM & Longformer")
        st.write(f"Number of overlapping documents: {len(overlap_mini_lm_longformer)}")
        for doc_id in overlap_mini_lm_longformer:
            st.markdown(format_doc_links(doc_id=doc_id) + "<br>", unsafe_allow_html=True) 

    with col2:
        st.subheader("MiniLM & MS Marco BERT")
        st.write(f"Number of overlapping documents: {len(overlap_mini_lm_msmarco_bert)}")
        for doc_id in overlap_mini_lm_msmarco_bert:
            st.markdown(format_doc_links(doc_id=doc_id), unsafe_allow_html=True) 
    
    with col3:
        st.subheader("Longformer & MS Marco BERT")
        st.write(f"Number of overlapping documents: {len(overlap_longformer_msmarco_bert)}")
        for doc_id in overlap_longformer_msmarco_bert:
            st.markdown(format_doc_links(doc_id=doc_id), unsafe_allow_html=True)

    with col4:
        st.subheader("All Models")
        st.write(f"Number of overlapping documents: {len(overlap_all_models)}")
        for doc_id in overlap_all_models:
            st.markdown(format_doc_links(doc_id=doc_id), unsafe_allow_html=True)

def find_doc_row(dataset: str, doc_id: str, *, as_dict=False, return_df=False):
    dataset_path = os.path.join("datasets", dataset)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"{dataset} not found")

    df_all = pd.read_csv(dataset_path, encoding="utf-8")
    try:
        row = df_all.loc[df_all["doc_id"] == doc_id].iloc[0]
    except IndexError:
        raise KeyError(f"{doc_id} not in {dataset}")

    df_row = row.to_frame().T
    df_row.insert(0, "View Doc", f"https://docviewer.history-lab.org/?doc_id={doc_id}")

    if return_df:
        return df_row
    if as_dict:
        return OrderedDict(df_row.iloc[0])
    return df_row.iloc[0]

def find_doc_text(dataset: str, doc_id: str) -> str:
    """
    Find the text of a document given its ID and dataset.
    """
    dataset_path = os.path.join("datasets", dataset)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset {dataset} does not exist.")
    
    with open(dataset_path, "r", encoding='utf-8') as f:
        reader = pd.read_csv(f)
        doc_text = reader.loc[reader["doc_id"] == doc_id, "body"].values[0]
    
    return doc_text

def convert_to_df(results: dict,dataset: str,  key: str = None) -> pd.DataFrame:
    """
    Build a DataFrame containing the *entire rows* from `dataset`
    that correspond to the doc IDs stored under `results[key]`.
    """
    if key is None:
        return pd.DataFrame([find_doc_row(dataset=dataset, doc_id=results)])

    doc_ids = results[key]                
    
    rows = [
        find_doc_row(dataset=dataset, doc_id=doc_id, as_dict=True)
        for doc_id in doc_ids
    ]
    
    return pd.DataFrame(rows).reset_index(drop=True)

def display_tables(results: dict, dataset: str) -> None:
    col1, col2, col3 = st.columns(3)

    with col1: 
        st.subheader("Model 1: MiniLM")
        st.write("Top 10 similar documents:")   
        st.dataframe(
            convert_to_df(results, key="mini_lm", dataset=dataset),
            hide_index=True,
            column_config={
                "View Doc": st.column_config.LinkColumn(
                    label="View",          
                    display_text="View"    
                )
            },
        )     

    
    with col2:
        st.subheader("Model 2: Longformer")
        st.write("Top 10 similar documents:")
        st.dataframe(
            convert_to_df(results, key="longformer", dataset=dataset),
            hide_index=True,
            column_config={
                "View Doc": st.column_config.LinkColumn(
                    label="View",          
                    display_text="View"    
                )
            },
        )    

    with col3:
        st.subheader("Model 3: MS Marco BERT")
        st.write("Top 10 similar documents:")
        st.dataframe(
            convert_to_df(results, key="msmarco_bert", dataset=dataset),
            hide_index=True,
            column_config={
                "View Doc": st.column_config.LinkColumn(
                    label="View",          
                    display_text="View"  
                )
            },
        )    

st.write("Randomly generate a document ID from a test set and view the retrieval results for the most similar documents.")
top_level_cols = st.columns([6, 4])
with top_level_cols[0]:
    st.write("* Test set v1: 10,000 documents randomly sampled with length 10-150 words.")
    st.write("* Test set v2: 10,000 documents randomly sampled with length 10-150 words.")
    st.write("* Test set v3: 10,000 documents randomly sampled with length 10-150 words.")
    st.write("* Test set v4: 4,201 documents randomly sampled with length 500-1000 words.")
    st.write("* Test set v5: 10,000 documents randomly sampled with length 100-400 words.")
with top_level_cols[1]:
    st.write("* MiniLM: all-MiniLM-L6-v2, 38-45 documents/sec")
    st.write("* Longformer: allenai/longformer-base-4096, 2-4 documents/sec")
    st.write("* MS Marco BERT: sentence-transformers/msmarco-bert-base-dot-v5, 8-14 documents/sec")

options = [f for f in os.listdir("datasets") if f.endswith(".csv")]
selected_option = st.selectbox("Choose an option:", options, key="random_doc_select")

if st.button("🔍 Find Similar Documents"):
    if selected_option:
        doc_id_search = random_doc_select(dataset=selected_option)
        try:
            main_doc_info = find_doc_row(dataset=selected_option, doc_id=doc_id_search)
            st.markdown(f"## Document: {main_doc_info['title'][:50]} ([*{doc_id_search}*]({main_doc_info['View Doc']}))")
            col = st.columns([7,3])
            with col[0]:
                st.write(f"{find_doc_text(doc_id=doc_id_search, dataset=selected_option)}")
                
            with col[1]:
                df = convert_to_df(results=doc_id_search, dataset=selected_option)

                if "View Doc" in df.columns:
                    df = df.drop(columns=["View Doc"])

                df_transposed = pd.DataFrame({
                    "Field": df.columns,
                    "Value": df.iloc[0].values
                })

                st.dataframe(df_transposed, hide_index=True)




            st.write("")
            st.write("")
            results = find_similar_docs(test_set=selected_option, doc_id=doc_id_search)
            display_tables(results=results, dataset=selected_option)
            display_stats(results=results)
        except Exception as e:
            st.write(f"Error: {e}")
            st.error("An error occurred while searching for similar documents. Please check the document ID and try again.")

# col1, col2, col3 = st.columns([3,1,9])
# random_doc_id = None

# with col1:
#     st.subheader("Randomly Generate Document")
#     st.write("Select a test set to randomly generate a document ID from that set.")
#     options = [f for f in os.listdir("datasets") if f.endswith(".csv")]
#     selected_option = st.selectbox("Choose an option:", options, key="random_doc_select")

#     col1_sub, col2_sub, col3_sub = st.columns([3, 7, 6])

#     with col2_sub:
#         if st.button("🎲 Generate Random DocID"):
#             random_doc_id = random_doc_select(dataset=selected_option)
    
#     if random_doc_id is not None:
#         st.write(f"*Randomly generated document ID:* **{random_doc_id}**")
#         st.code(random_doc_id, language="text")

# with col3:
#     st.subheader("Search for Similar Documents")
#     st.write("Enter a document ID to find similar documents across all 3 models.")
#     st.write("")
    
    
#     options_search = [f for f in os.listdir("datasets") if f.endswith(".csv")]
#     selected_option_search = st.selectbox("Choose an option:", options_search, key="search_sim_docs")
#     doc_id_search = st.text_input("Document ID", key="doc_id_input")
    
#     if st.button("🔍 Find Similar Documents"):
#         if doc_id_search and selected_option_search:
#             try:

#                 st.markdown(f"## Document: *{doc_id_search}* Info")
#                 st.write(f"{find_doc_text(doc_id=doc_id_search, dataset=selected_option_search)}")
#                 st.write("")
#                 st.write("")
#                 results = find_similar_docs(test_set=selected_option_search, doc_id=doc_id_search)
#                 display_tables(results=results, dataset=selected_option_search)
#                 display_stats(results=results)
#             except Exception as e:
#                 st.write(f"Error: {e}")
#                 st.error("An error occurred while searching for similar documents. Please check the document ID and try again.")
#         else:
#             st.warning("Please enter a document ID / test dataset to search.")