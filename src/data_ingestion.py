import trafilatura
from typing import List
import nltk
from nltk.tokenize import sent_tokenize
import os

class DataIngestion:
    def __init__(self):
        self.paper_url = "https://arxiv.org/abs/1706.03762"
        # Set up NLTK data path
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        # Initialize NLTK
        nltk.data.path.append(nltk_data_dir)
        try:
            # Download punkt tokenizer data
            nltk.download('punkt', download_dir=nltk_data_dir)
        except Exception as e:
            print(f"Error downloading NLTK data: {e}")
            # Fallback to basic sentence splitting
            print("Using fallback tokenization method")

    def download_paper(self) -> str:
        """Download and extract text content from the paper."""
        downloaded = trafilatura.fetch_url(self.paper_url)
        text = trafilatura.extract(downloaded)
        return text

    def create_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []

        try:
            # Try using NLTK's sentence tokenizer
            sentences = sent_tokenize(text)
        except:
            # Fallback to basic sentence splitting
            print("Falling back to basic sentence splitting")
            sentences = [s.strip() for s in text.split('.') if s.strip()]

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > chunk_size:
                # Store the current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Keep overlap sentences for the next chunk
                overlap_sentences = current_chunk[-3:]  # Keep last 3 sentences
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks