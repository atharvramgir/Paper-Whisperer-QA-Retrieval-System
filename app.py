# Contents of app.py
import streamlit as st
import json
import pandas as pd
from src.data_ingestion import DataIngestion
from src.embeddings import EmbeddingManager
from src.rag_pipeline import RAGPipeline
from src.evaluation import Evaluator

# Initialize components
@st.cache_resource
def init_components():
    data_ingestion = DataIngestion()
    embedding_manager = EmbeddingManager()
    rag_pipeline = RAGPipeline()
    evaluator = Evaluator()
    return data_ingestion, embedding_manager, rag_pipeline, evaluator

# Initialize the paper content
@st.cache_data
def load_paper_content():
    data_ingestion, embedding_manager, _, _ = init_components()
    paper_text = data_ingestion.download_paper()
    chunks = data_ingestion.create_chunks(paper_text)
    embedding_manager.generate_embeddings(chunks)
    return chunks

# Main app
st.title("Attention Is All You Need - Q&A System")
st.markdown("""
This Q&A system allows you to ask questions about the 'Attention Is All You Need' paper. 
The system uses RAG (Retrieval-Augmented Generation) to provide accurate answers with citations.
""")

# Initialize components
data_ingestion, embedding_manager, rag_pipeline, evaluator = init_components()

# Load paper content
with st.spinner("Loading paper content..."):
    chunks = load_paper_content()

# Query input
query = st.text_input("Enter your question about the paper:", 
                      "What is the main contribution of this paper?")

if st.button("Get Answer"):
    with st.spinner("Searching and generating response..."):
        # Retrieve relevant contexts
        contexts = embedding_manager.search_similar(query)

        # Generate response
        response = rag_pipeline.generate_response(query, contexts)
        response_dict = json.loads(response)

        # Evaluate retrieval
        metrics = evaluator.evaluate_retrieval(query, contexts, response_dict["answer"])

        # Display response
        st.markdown("### Answer")
        st.write(response_dict["answer"])

        st.markdown("### Citations")
        st.write(response_dict["citations"])

        # Display retrieval metrics
        st.markdown("### Retrieval Metrics")
        metrics_df = pd.DataFrame([metrics])
        st.dataframe(metrics_df.style.format({
            'avg_context_length': '{:.0f}',
            'avg_term_overlap': '{:.2f}'
        }))

        # Show retrieved contexts
        with st.expander("View Retrieved Contexts"):
            for i, context in enumerate(contexts, 1):
                st.markdown(f"**Context {i}:**")
                st.write(context)