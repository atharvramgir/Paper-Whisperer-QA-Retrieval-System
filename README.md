## To run the project locally:

1. Install dependencies:
```bash
pip install chromadb nltk openai pandas scikit-learn streamlit trafilatura
```

2. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key'
```

3. Create the project structure:
   - Create the main directory for your project
   - Create a `src` directory inside it
   - Create a `.streamlit` directory
   - Copy each file to its respective location as shown in the structure above

4. Run the application:
```bash
streamlit run app.py
```

The application will be available at http://localhost:5000


## Technical Implementation

### How Retrieval and Generation Work

The system implements a RAG (Retrieval-Augmented Generation) pipeline with the following components:

1. **Data Ingestion and Chunking**
   - The paper is downloaded using Trafilatura for clean text extraction
   - Text is split into overlapping chunks (1000 chars with 200 char overlap)
   - NLTK's sentence tokenizer ensures context-aware chunking
   - Fallback mechanisms handle potential tokenization failures

2. **Retrieval System**
   - Uses TF-IDF vectorization for creating text embeddings
   - ChromaDB manages vector storage and similarity search
   - When a query is received:
     - Query is converted to TF-IDF vector
     - Vector similarity search finds relevant chunks
     - Top 3 most similar chunks are retrieved

3. **Generation Process**
   - Retrieved contexts are combined with the user query
   - Structured prompt ensures consistent output format
   - GPT-4 generates responses with:
     - Direct answers based on context
     - Relevant citations from the paper
     - JSON-formatted output for consistent parsing

4. **Quality Metrics**
   - Tracks retrieval performance through:
     - Context length analysis
     - Query-context term overlap
     - Number of relevant contexts used

## Features
- Interactive Streamlit web interface
- Automatic paper content retrieval
- Context-aware response generation
- Citation support
- Retrieval quality metrics
- Interactive context viewer

## Project Structure
```
.
├── app.py                 # Main Streamlit application
├── src/
│   ├── data_ingestion.py # Paper downloading and chunking
│   ├── embeddings.py     # Vector embeddings and search
│   ├── evaluation.py     # Retrieval metrics
│   └── rag_pipeline.py   # OpenAI integration
└── .streamlit/
    └── config.toml       # Streamlit configuration