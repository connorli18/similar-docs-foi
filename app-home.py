import os
import streamlit as st
from st_helper import random_doc_select, find_similar_docs
import pandas as pd

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
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

def format_doc_links(doc_id, base_url="https://connorli18.pythonanywhere.com/articlespec/primeprog3k2"):
    #return f"<a href='{base_url}{doc_id}' target='_blank'>{doc_id}</a>"
    return f"<a href='{base_url}' target='_blank'>{doc_id}</a>"


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

def convert_to_df(results: dict, key: str, dataset: str) -> pd.DataFrame:

    return pd.DataFrame ({
        "doc_id": results[key],
        "text_preview": [find_doc_text(doc_id=doc_id, dataset=dataset) for doc_id in results[key]],
    })

def display_tables(results: dict, dataset: str) -> None:
    col1, col2, col3 = st.columns(3)

    with col1: 
        st.subheader("Model 1: MiniLM")
        st.write("Top 10 similar documents:")        
        st.dataframe(convert_to_df(results=results, key="mini_lm", dataset=dataset).reset_index(drop=True))

    
    with col2:
        st.subheader("Model 2: Longformer")
        st.write("Top 10 similar documents:")
        st.dataframe(convert_to_df(results=results, key="longformer", dataset=dataset).reset_index(drop=True))

    with col3:
        st.subheader("Model 3: MS Marco BERT")
        st.write("Top 10 similar documents:")
        st.dataframe(convert_to_df(results=results, key="msmarco_bert", dataset=dataset).reset_index(drop=True))



col1, col2, col3 = st.columns([3,1,9])
random_doc_id = None

with col1:
    st.subheader("Randomly Generate Document")
    st.write("Select a test set to randomly generate a document ID from that set.")
    options = [f for f in os.listdir("datasets") if f.endswith(".csv")]
    selected_option = st.selectbox("Choose an option:", options, key="random_doc_select")

    col1_sub, col2_sub, col3_sub = st.columns([3, 7, 6])

    with col2_sub:
        if st.button("🎲 Generate Random DocID"):
            random_doc_id = random_doc_select(dataset=selected_option)
    
    if random_doc_id is not None:
        st.write(f"*Randomly generated document ID:* **{random_doc_id}**")
        st.code(random_doc_id, language="text")

with col3:
    st.subheader("Search for Similar Documents")
    st.write("Enter a document ID to find similar documents across all 3 models.")
    st.write("")
    
    
    options_search = [f for f in os.listdir("datasets") if f.endswith(".csv")]
    selected_option_search = st.selectbox("Choose an option:", options_search, key="search_sim_docs")
    doc_id_search = st.text_input("Document ID", key="doc_id_input")
    
    if st.button("🔍 Find Similar Documents"):
        if doc_id_search and selected_option_search:
            try:

                st.markdown(f"## Document: *{doc_id_search}* Info")
                st.write(f"{find_doc_text(doc_id=doc_id_search, dataset=selected_option_search)}")
                st.write("")
                st.write("")
                results = find_similar_docs(test_set=selected_option_search, doc_id=doc_id_search)
                display_tables(results=results, dataset=selected_option_search)
                display_stats(results=results)
            except:
                st.error("An error occurred while searching for similar documents. Please check the document ID and try again.")
        else:
            st.warning("Please enter a document ID / test dataset to search.")
